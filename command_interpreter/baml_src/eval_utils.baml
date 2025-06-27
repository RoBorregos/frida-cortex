
function GenerateEnrichedAndReorderedCommand(request: string) -> string {
  client GEMINI_FLASH_2_5
  prompt #"
    You will receive a natural language command to be executed by a robot.
    Your task is to both paraphrase AND reorder the command while keeping the exact same intent and task objectives.
    
    Steps to follow:
    1. Paraphrase the command using synonyms for actions while maintaining the same meaning
    2. Change the word order to create a different sentence structure
    3. Keep all the same information - do not add or remove any details
    
    For example, if the command is:
    'go to the living room then locate a knife and get it and bring it to Angel in the bathroom',
    the result could be:
    'Navigate toward Angel in the bathroom after proceeding to the living room to find and collect a knife'
    
    Another example:
    'Pick up the red cup from the kitchen counter'
    could become:
    'From the kitchen counter, collect the red cup'
    
    Just return one transformed command in one line.
    {{ request }}
  "#
}

test GenerateEnrichedAndReorderedCommandTest {
  functions [GenerateEnrichedAndReorderedCommand]
  args {
    request "look for a toy in the office then get it and put it on the kitchen table"
  }
}