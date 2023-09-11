pipenv run python eval.py --prompt_path prompts/gpt_annotate_example.txt --mode annotate --dataset norsynthclinical --model gpt-chat --modelName gpt-4 --openAIKey $Env:OPENAI_API_KEY --output results/gpt-4-annotate-norsynth.json
pipenv run python eval.py --prompt_path prompts/gpt_replace_example.txt --mode replace --dataset norsynthclinical --model gpt-chat --modelName gpt-4 --openAIKey $Env:OPENAI_API_KEY --output results/gpt-4-replace-norsynth.json
pipenv run python eval.py --prompt_path prompts/gpt_annotate_example.txt --mode annotate --dataset datasets/nor-deid-synthdata/holdout.spacy --model gpt-chat --modelName gpt-4 --openAIKey $Env:OPENAI_API_KEY --output results/gpt-4-annotate-synthdata.json
pipenv run python eval.py --prompt_path prompts/gpt_replace_example.txt --mode replace --dataset datasets/nor-deid-synthdata/holdout.spacy --model gpt-chat --modelName gpt-4 --openAIKey $Env:OPENAI_API_KEY --output results/gpt-4-replace-synthdata.json
pipenv run python eval.py --prompt_path prompts/gpt_annotate_example.txt --mode annotate --dataset norsynthclinical --model gpt-chat --modelName gpt-3.5-turbo --openAIKey $Env:OPENAI_API_KEY --output results/gpt-3.5-annotate-norsynth.json
pipenv run python eval.py --prompt_path prompts/gpt_replace_example.txt --mode replace --dataset norsynthclinical --model gpt-chat --modelName gpt-3.5-turbo --openAIKey $Env:OPENAI_API_KEY --output results/gpt-3.5-replace-norsynth.json
pipenv run python eval.py --prompt_path prompts/gpt_annotate_example.txt --mode annotate --dataset datasets/nor-deid-synthdata/holdout.spacy --model gpt-chat --modelName gpt-3.5-turbo --openAIKey $Env:OPENAI_API_KEY --output results/gpt-3.5-annotate-synthdata.json
pipenv run python eval.py --prompt_path prompts/gpt_replace_example.txt --mode replace --dataset datasets/nor-deid-synthdata/holdout.spacy --model gpt-chat --modelName gpt-3.5-turbo --openAIKey $Env:OPENAI_API_KEY --output results/gpt-3.5-replace-synthdata.json
