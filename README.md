# fs rule questions
This repository contains a experimental python script to generate questions about each rule in the EV section of the fsg. The generation was done with the huggingface library and [potsawee's model ("t5-large-generation-squad-QuestionAnswer")](https://huggingface.co/potsawee/t5-large-generation-squad-QuestionAnswer).  
The script contains functions to extract text from the fsg rules pdf and to convert that text into a json format. Some manual preprocessing is required for the conversion to json. This is further explained in the script.  

The generated question can be found in questions.txt
*Most questions are not very usefull unless you want to memorize the ruleset exactly.*
