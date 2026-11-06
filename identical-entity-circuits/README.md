# Finding Duplicate Detector Circuits in GPT-2

![Identical Entity Circuits](./download.png)

This experiment is based on finding the crucial circuits which understands whether the entities in a sentence belongs to single or different. For example, 

`John gave bag to John`

This prompt will help us to understand which circuit of GPT-2 will distinguish the identical entities and how classifies. To investigate it further, I added another line of prompt to test this:

`John gave bag to John: Are the recipient and the giver same?`

Final Tokens: `yes` or `no`  
