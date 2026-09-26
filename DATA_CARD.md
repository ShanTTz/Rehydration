# Data Card

## Bundled Data

The package includes selected Reddit discussions from five communities:
AskReddit, aww, funny, science, and worldnews. It contains 500 selected posts,
filtered comments, author-history features, frozen intent pools, and derived
cascade summaries. The source dataset reconstructs 482 discussion cascades.

Author labels across CSV tables and agent-profile usernames were replaced
with consistent, release-specific pseudonyms. Profile real-name fields were
neutralized. The salt and reverse mapping are not included. Original post and
comment IDs, timestamps, URLs, and text may still link records to public
discussions; this is pseudonymization, not a guarantee of anonymity.

## External Inputs

Large Hacker News, Lemmy, TBBT, and Reddit expansion collections are not
bundled. Data collection and normalization code is provided, but access to
those analyses requires the corresponding source data and any necessary API
access. Private participant-level human-study records are excluded; only
aggregate research outputs belong in a public release.

## Responsible Use

The included material contains user-generated public discussion text. Review
the terms and permissions applicable to the upstream platform and source
dataset before redistributing or using the text outside this research context.
Behavioral features describe observed discussion signals and are not clinical
or personality diagnoses.
