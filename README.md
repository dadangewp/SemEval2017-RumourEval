This system was built to deal with SemEval-2017 task 3 SubTask A. To be able to run this system, you need to configure several "stuffs".

Requirement
- numpy
- scipy
- scikit-learn
- nltk
- VADER

Usage

1. Download the dataset on the SemEval-2017 website (http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools).

2. Download several affective resources that used in this systems.
- Emolex (http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- EmoSenticNet (https://www.gelbukh.com/emosenticnet/)
- LIWC (http://www.liwc.net/download.php)
- DAL Normalized (Provided in this project)
- ANEW Normalized (Provided in this project)
- AFINN Normalized (Provided in this project)

3. Resolve several paths of file.
- Several paths in iodata folder.
- Paths of affective resources.

4. Run the program by executing main class.