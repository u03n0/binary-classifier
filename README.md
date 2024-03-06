# binary-classifier
classifying if a companys 'about' section on LinkedIn is related to sustainability and environmental impact.

## How to run:
in order to use this with Streamlit, you need to have `docker`.
- clone the repo
- in the command line `docker build -t <name> .` replace `<name>` with whatever you want.
- check once its been created with `docker images`
- to run the container use `docker run -p 8501:8501 <name>` again replacing `<name>` with the original name you gave this image.
- click on the url link displayed in the terminal.

**warning** building the `docker image` will take some time, as will training the model for the first time (~8 minutes) with a GPU.
