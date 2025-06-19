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

        And the user instruction:
        {{ user_instruction }}

        Return a concise and short response to the user covering all the
        aspects of the user instruction. Don't include the question in the response.

        The response should be in the same language as the user instruction.
    "#
}