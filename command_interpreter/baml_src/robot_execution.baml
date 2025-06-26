class TaskExecutionResult {
    action string
    success bool
    result string
}

function AugmentedResponse(relevant_document: string, user_instruction: string) -> string{
    client GEMINI_FLASH_2_5
    prompt #"
        You are a service robot assistant, the user has asked you a question based
        on a previously executed command or general knowledge.

        Based on this relevant information:
        {{ relevant_document }}

        And the user instruction, it might contain previous actions already executed,
        that don't need to be mentioned in the final response:
        {{ user_instruction }}

        For example, if the user instruction is:
        'go to the bed get the name of the person there, and tell it back to me'
        And the relevant document is:
        '[{'document': 'get_person_info', 'metadata': {'context': '', 'original_name': 'get_person_info', 'result': 'person name is John', 'timestamp': '2025-06-23T00:13:40.147798', 'status': 1, 'shelve': '', 'command': "action='get_person_info' info_type='name'"}}]

        The response should be:
        'John is the name of the person in the bed'

        Return a concise and short response to the user covering all the
        aspects of the user instruction. Don't include the question in the response.

        The response should be in the same language as the user instruction.
    "#
}