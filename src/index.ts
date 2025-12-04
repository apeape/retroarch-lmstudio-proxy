import express from "express";
import * as PImage from "pureimage"
import { bytesToBase64 } from 'byte-base64'
import { WritableStreamBuffer } from 'stream-buffers'
import { LLM, LMStudioClient } from "@lmstudio/sdk";
import Ocr from '@gutenye/ocr-node';
import pino from "pino";
import fs from "fs/promises";
import os from "os";
import path from "path";
import natural from "natural";

const app = express()
const port = 4404

app.use(express.json({
    type: () => true
}))

const font = PImage.registerFont("fonts/Mplus1Code-Bold.otf", "MPlus1Code")
const font2 = PImage.registerFont("fonts/ChiKareGo2.ttf", "ChiKareGo2")
const MAX_LINE_LENGTH = 60;

const lmstudio = new LMStudioClient();
var model: LLM

const MODEL_NAME = "huihui-minicpm-v-4_5-abliterated";
const OUTPUT_WIDTH = 640*3
const OUTPUT_HEIGHT = 400*3
let prev_x = 40
let prev_y = OUTPUT_HEIGHT - 100;

const logger = pino({
    level: process.env.PINO_LOG_LEVEL || 'debug',
    timestamp: pino.stdTimeFunctions.isoTime,
  });

app.post('/', async (req, res) => {
    const input = req.body

    logger.debug(req, "request")

    const llmFile = await lmstudio.files.prepareImageBase64("screenshot.png", input.image)

    let buf = Buffer.from(input.image, "base64");

    // write to temp PNG
    
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "ra-ocr-"));
    const tmpPath = path.join(tmpDir, "frame.png");
    await fs.writeFile(tmpPath, buf);

    const { width: inputWidth, height: inputHeight } = getPngSize(buf);

    const assetsDir = path.resolve(__dirname, "..", "assets");

    const ocr = await Ocr.create({
        models: {
            detectionPath: path.join(assetsDir, "ch_PP-OCRv4_det_infer.onnx"),
            recognitionPath: path.join(assetsDir, "ch_PP-OCRv4_rec_infer.onnx"),
            dictionaryPath: path.join(assetsDir, "ppocr_keys_v1.txt"),
        }
    });
    const ocrRes = await ocr.detect(tmpPath);

    const { ocrTexts } = ocrRes; // array of TextLine
    //logger.debug(ocrRes)

    fs.rm(tmpDir, { recursive: true, force: true }, (err) => {
        if (err) {
            console.error(`Error deleting folder: ${err.message}`);
            return;
        }
        console.log(`Folder "${tmpDir}" and its contents deleted successfully.`);
    });

    const result = await model.respond({
        content: `You're an expert translator. Extract the texts in the image and translate them. Character names or the dialogue may be surrounded in brackets like this: 「Name」. Multiple lines of dialogue in the same location should be merged into one entry with consecutive sentences. Format output as JSON. Remember to escape quotes and other characters requires by JSON spec:
        json\`\`\`
        [
            {
                "location": "message window",
                "original": "「キヨミ」 こんにちは。",
                "originalLanguage": "Japanese",
                "translation": "'Kiyomi': Hello.",
                "translationLanguage": "English"
            },
            {
                "location": "left side menu",
                "original": "USE\nGO!!\nLOAD\nSAVE\nMORE"
                "originalLanguage": "English",
                "translation": "USE\nGO!!\nLOAD\nSAVE\nMORE",
                "translationLanguage": "English"
            },
        ]
        \`\`\`
        `,
        images: [llmFile]
    }, {
        structured: {
            type: "json",
            jsonSchema: translationListJsonSchema,
        },
        maxTokens: 2000
    });  

    let parsed = JSON.parse(result.content);

    logger.debug(result.content, "result from LM Studio")

    logger.debug(parsed, "parsedJSON")

    let translations = parsed;
    const outImg = PImage.make(OUTPUT_WIDTH, OUTPUT_HEIGHT);

    // get canvas context
    const ctx = outImg.getContext("2d");

    ctx.clearRect(0, 0, outImg.width, outImg.height)

    const lineHeight = 56;
    //cleanup translation list
    let count = translations.length;
    translations = translations.filter(tr => tr.originalLanguage != tr.translationLanguage);
    translations = filterSimilarItems(translations, (a, b) => natural.JaroWinklerDistance(a, b), 0.7);
    logger.debug(`Removed ${count - translations.length}/${count} dupes / untranslated entries`)
    for (const t of translations) {

        let segment = findBestOCRSegment(t.original, ocrRes);
        let ocrX = prev_x;
        let ocrY = prev_y;

        if (segment == null) logger.debug("OCR error, reusing previous coordinates")
        else
        {
          logger.debug(`Found OCR segment: ${segment.box[0][0]} ${segment.box[0][1]}`);
          logger.debug(`OCR text: ${segment.text}`);
          const { x: topLeftX, y: topLeftY } = scalePointAR(
            segment.box[0][0], 
            segment.box[0][1],
            inputWidth, inputHeight,
            OUTPUT_WIDTH, OUTPUT_HEIGHT,
            30 / 17
          );
          ocrX = topLeftX;
          prev_x = topLeftX;
          ocrY = topLeftY;
          prev_y = topLeftY;

          const { x: bottomRightX, y: bottomRightY } = scalePointAR(
              segment.box[2][0], 
              segment.box[2][1],
              inputWidth, inputHeight,
              OUTPUT_WIDTH, OUTPUT_HEIGHT,
              30 / 17
          );

          ctx.fillStyle = "#000000de"
          ctx.fillRect(ocrX - 20, ocrY - 40, (bottomRightX + 20) - ocrX, 
            (bottomRightY - (ocrY - 40)) * 2)

          logger.debug(`scaled OCR X: ${ocrX} Y: ${ocrY}`)
        }
        logger.debug(`LLM translation: ${t.translation}`);


        let yPos = ocrY;
        let xPos = ocrX;
        ctx.font = `${lineHeight}pt ChiKareGo2`

        const lines = t.translation.split('\n')

        for (const l of lines) {
            const wrappedLines = wrapLineWordwise(l, MAX_LINE_LENGTH);

            for (const s of wrappedLines) {
                const text = s.trim().replaceAll("…", ".");
                let metrics = ctx.measureText(text);

                ctx.fillStyle = "#000000de"
                ctx.fillRect(xPos - 20, yPos - 40, metrics.width, lineHeight)

                ctx.fillStyle = "cyan";
                ctx.strokeStyle = '#000000'; // Outline color (black)
                ctx.lineWidth = 12;
                ctx.strokeText(text, xPos, yPos);
                ctx.fillText(text, xPos, yPos);
                yPos += lineHeight;
            }
        }

    }

    const stream = new WritableStreamBuffer()

    await PImage.encodePNGToStream(outImg, stream)

    const content = stream.getContents()
    if (!content) {
        res.json({
            error: "failed to create image."
        })

        return
    }
    res.json({
        image: bytesToBase64(content)
    })
})

function wrapLineWordwise(line, maxLen = MAX_LINE_LENGTH) {
  const words = line.split(/\s+/);
  const out = [];
  let current = '';

  for (let word of words) {
    if (!word) continue;

    // If the word is longer than maxLen, hyphenate it
    while (word.length > maxLen) {
      const chunk = word.slice(0, maxLen - 1) + '-'; // "-” makes it maxLen
      if (current) {
        out.push(current);
        current = '';
      }
      out.push(chunk);
      word = word.slice(maxLen - 1);
    }

    if (!current) {
      // Start a new line
      current = word;
    } else if (current.length + 1 + word.length <= maxLen) {
      // Fits on current line with a space
      current += ' ' + word;
    } else {
      // Doesn't fit, push current and start new line
      out.push(current);
      current = word;
    }
  }

  if (current) out.push(current);

  return out;
}

function findBestMatchIterative(items, scoreFn) {
    if (!items || items.length === 0) {
        return null;
    }

    let bestMatch = items[0];
    let highestScore = scoreFn(items[0]);

    for (let i = 1; i < items.length; i++) {
        const currentItem = items[i];
        const currentScore = scoreFn(currentItem);

        if (currentScore > highestScore) {
            highestScore = currentScore;
            bestMatch = currentItem;
        }
    }

    return bestMatch;
}

function findBestOCRSegment(str, ocrSegments) {
    return findBestMatchIterative(ocrSegments, ocr => 
        natural.JaroWinklerDistance(str, ocr.text)); // 0-1
}

function getPngSize(buffer) {
  // PNG signature occupies 8 bytes, IHDR chunk begins immediately after.
  // IHDR data: width (4 bytes), height (4 bytes), both big-endian.

  if (buffer.readUInt32BE(0) !== 0x89504e47) {
    throw new Error("Not a PNG file");
  }

  const width  = buffer.readUInt32BE(16);
  const height = buffer.readUInt32BE(20);

  return { width, height };
}

function scalePoint(x, y, inWidth, inHeight, outWidth, outHeight) {
  return {
    x: x * (outWidth / inWidth),
    y: y * (outHeight / inHeight)
  };
}

/**
 * Scale a point from an input resolution into a pillarboxed/letterboxed
 * region centered within an output canvas.
 *
 * @param {number} x - input X coordinate
 * @param {number} y - input Y coordinate
 * @param {number} inWidth - input logical width
 * @param {number} inHeight - input logical height
 * @param {number} outWidth - full output canvas width
 * @param {number} outHeight - full output canvas height
 * @param {number} targetAspect - desired aspect ratio (width/height), e.g. 4/3 for 4:3
 */
function scalePointAR(
  x,
  y,
  inWidth,
  inHeight,
  outWidth,
  outHeight,
  targetAspect
) {
  // Compute the size of the centered viewport that preserves targetAspect
  const outAspect = outWidth / outHeight;

  let viewWidth, viewHeight, offsetX, offsetY;

  if (outAspect > targetAspect) {
    // Output is wider than target: full height used, pillarbox left/right
    viewHeight = outHeight;
    viewWidth = viewHeight * targetAspect;
    offsetX = (outWidth - viewWidth) / 2;
    offsetY = 0;
  } else {
    // Output is taller/narrower than target: full width used, letterbox top/bottom
    viewWidth = outWidth;
    viewHeight = viewWidth / targetAspect;
    offsetX = 0;
    offsetY = (outHeight - viewHeight) / 2;
  }

  // Uniform scale from input coordinates into the inner viewport
  const scaleX = viewWidth / inWidth;
  const scaleY = viewHeight / inHeight;

  // If your input coordinates are already within [0, inWidth] x [0, inHeight],
  // this will guarantee they stay inside the centered game area.
  const outX = offsetX + x * scaleX;
  const outY = offsetY + y * scaleY;

  return { x: outX, y: outY };
}

/**
 * Filters out items from a list that are too similar to previously encountered items.
 *
 * @param {Array<any>} list The input list of items.
 * @param {Function} similarityScoringFunction A function that takes two items as input
 *   and returns a similarity score (e.g., a number between 0 and 1, where 1 is identical).
 * @param {number} threshold The minimum similarity score for two items to be considered "repeated".
 * @returns {Array<any>} A new list containing only the unique items based on the similarity threshold.
 */
function filterSimilarItems(list, similarityScoringFunction, threshold = 0.7) {
  if (!Array.isArray(list)) {
    throw new Error("Input 'list' must be an array.");
  }
  if (typeof similarityScoringFunction !== 'function') {
    throw new Error("Input 'similarityScoringFunction' must be a function.");
  }
  if (typeof threshold !== 'number') {
    throw new Error("Input 'threshold' must be a number.");
  }

  const uniqueItems = [];

  for (const item of list) {
    let isSimilarToExisting = false;
    for (const uniqueItem of uniqueItems) {
      if (similarityScoringFunction(item, uniqueItem) >= threshold) {
        isSimilarToExisting = true;
        break; // Found a similar item, no need to check further
      }
    }
    if (!isSimilarToExisting) {
      uniqueItems.push(item);
    }
  }

  return uniqueItems;
}

export const translationEntryJsonSchema = {
  type: "object",
  properties: {
    location: { type: "string" },

    original: { type: "string" },

    originalLanguage: { type: "string" },

    translation: { type: "string" },

    translationLanguage: { type: "string" }
  },
  required: [
    "location",
    "original",
    "originalLanguage",
    "translation",
    "translationLanguage"
  ],
  additionalProperties: false
} as const;

export const translationListJsonSchema = {
  type: "array",
  items: translationEntryJsonSchema
} as const;

app.listen(port, async () => {
    await font.load()
    await font2.load()
    model = await lmstudio.llm.model(MODEL_NAME);

    logger.info(`Server started at port ${port}.`)
})
