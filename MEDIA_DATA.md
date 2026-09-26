# Media Data

The original collection contained Reddit media caches under
community folders such as `aww_data/reddit_media` and
`worldnews_data/reddit_media`.

Those files are downloaded post images or video thumbnails collected by the old
`reddit_data_collector.py` script. The collector stores the local file path in
the post CSV column `media_saved_path`.

## Current Availability

The restored machine no longer contains the `social.zip`, `reddit数据1.zip`,
or `reddit数据2.zip` archives listed in `data/source_archives_inventory.json`.
The repository therefore cannot bundle the binary images from those archives.
The inventory and every `media_saved_path` value are retained so a future
archive restore can materialize exactly the selected assets.

## Why Images Are Not Required For Current Results

The current paper reproduction pipeline does not read image files. It uses:

- Post title and body text.
- Early comments.
- Author history summaries.
- Enriched text/behavior features.
- Empirical cascade statistics.
- Frozen semantic intents.
- Community population and calibration files.

The old LLM feature extractor also analyzes text fields and comment logs; it
does not load local images from `media_saved_path`.

The current structural and semantic replay results do not load image pixels.
They remain fully reproducible from the bundled text and interaction data.

## Selected Paper Data Media Coverage

The selected 500 paper posts keep the original `media_saved_path` column for
traceability:

- `AskReddit`: 0 of 100 selected posts have cached media paths.
- `aww`: 54 of 100 selected posts have cached media paths.
- `funny`: 84 of 100 selected posts have cached media paths.
- `science`: 86 of 100 selected posts have cached media paths.
- `worldnews`: 85 of 100 selected posts have cached media paths.

These paths point back to the original raw `social` collection and are not used
by the local simulator.

## When To Add Images Back

Add the media files only if extending the project to a multimodal version of the
paper, for example:

- Visual virality features.
- Meme or image-content classification.
- Image-conditioned intent generation.
- Multimodal LLM feature extraction.

In that case, copy only the selected posts' media into each community's
`reddit_media` folder and update `media_saved_path` to repository-relative
paths.
