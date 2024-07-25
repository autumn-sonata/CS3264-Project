To run prediction on a single URL, running in commandline:
1. get `rf-general.pkl`, `rf-minimal.pkl`, `rf-lexical.pkl` from directories with the same model name at root level.
2. Run in command line:
```
python3 prediction.py URL
```
Where URL is the URL you want to run prediction on.
For example:
- `python3 prediction.py https://www.google.com/`
- `python3 prediction.py https://www.facebook.com/`
