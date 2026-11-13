in this commit i have added chat history using MessagePlaceHoldder . This is passed into the prompt.invoke .
We dont need to make it part of the 
system_prompt=""

""
varible . It is added when prompt.invoke is called . 