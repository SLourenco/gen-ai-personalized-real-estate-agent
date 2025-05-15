# gen-ai-personalized-real-estate-agent
Final project for the Udacity Building Generative AI Solutions course

## Setup

After creating a virtual environment, to install requirements: 
```bash
pip install -r requirements.txt
```

The script reads configuration values from the environment.
For example, you can create a local environment file `.env` with the following variables:

```
OPENAI_API_KEY="YOUR API KEY"
```

Available environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `GENERATE_LISTINGS`: if `True`, generates new listings instead of using the existing ones.

## Run the app
To run the app, use the following command:
```bash
python HomeMatch.py
```
