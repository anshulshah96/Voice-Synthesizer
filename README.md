Voice Synthesizer
=====================================

Setup
=====================================

1. Clone the Repository.
```shell
git clone https://github.com/anshulshah96/Voice-Synthesizer/
```

2. Install [Anaconda](https://www.continuum.io/downloads#linux). Preferably Python 2.7.

3. Create Conda Environment
```bash
conda create --name voice python=2.7
source activate voice
```

4. Install Requirements
```bash
sudo apt-get install python-pyaudio libopenblas-dev
pip install numpy scipy matplotlib sklearn pandas tables Theano
conda install nb_conda
```

5. In ~/.keras/keras.json change tensorflow to theano

License
=========
[MIT License](https://anshul.mit-license.org/)
