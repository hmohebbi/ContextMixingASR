# Context Mixing in Speech Transformers

The official repo for the [EMNLP 2023](https://2023.emnlp.org/) paper "__Homophone Disambiguation Reveals Patterns of Context Mixing in Speech Transformers__"

🤗[[Data]](https://huggingface.co/datasets/hosein-m/french_homophone_asr)

📃[[Paper]](https://aclanthology.org/2023.emnlp-main.513.pdf)

## Abstract
> Transformers have become a key architecture in speech processing, but our understanding of how they build up representations of acoustic and linguistic structure is limited.  In this study, we address this gap by investigating how measures of `context-mixing' developed for text models can be adapted and applied to models of spoken language.  We identify a linguistic phenomenon that is ideal for such a case study: homophony in French (e.g. livre vs livres), where a speech recognition model has to attend to syntactic cues such as determiners and pronouns in order to disambiguate spoken words with identical pronunciations and transcribe them while respecting grammatical agreement. We perform a series of controlled experiments and probing analyses on Transformer-based speech  models. Our findings reveal that representations in encoder-only models effectively incorporate these cues to identify the correct transcription, whereas encoders in encoder-decoder models mainly relegate the task of capturing contextual dependencies to decoder modules.
