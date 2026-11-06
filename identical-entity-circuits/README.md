# Finding Duplicate Detector Circuits in GPT-2

![Identical Entity Circuits](./download.png)

This experiment is based on finding the crucial circuits which understands whether the entities in a sentence belongs to single or different. For example, 

`John gave bag to John`

This prompt will help us to understand which circuit of GPT-2 will distinguish the identical entities and how classifies. To investigate it further, I added another line of prompt to test this:

`John gave bag to John: Are the recipient and the giver same?`

Final Tokens: `yes` or `no`

# Purpose of Testing

Initially, the goal was just to look at the circuits which is responsible for information mover. Because, the expectations was to predict `yes` but in reality the output should favor `no`.
You can think why `no` is the responsible token or final token. That's what leaded to this experiment how does the duplicate heads favor to the final token `yes` which has 2 factors:
- Changing probabilities for the final token
- Essentially the Model needs to think _might be there are 2 person_

# Safety and Alignment

After you run this, don't worry I have added the notebook which does help you to test the explanations which has written. The Safety question could be:
- Model successfully distinguish 2 different person having same entity _Alignment_
- If the Model is confident, are they conscious? _Safety_

Please feel free to think over it, see the notebook couple of times. I would love to have your thoughts on it. Because all of these are assumptions not the final discovery.
