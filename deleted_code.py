
    

    # def book_chat_completion(client, model, user_question, relevant_excerpts):
    #     chat_completion = client.chat.completions.create(
    #         messages=[
    #             # {
    #             #     "role": m["system"],
    #             #     "content": m["You are a book expert. Given the user's question and relevant excerpts from a book, answer the question by including direct quotes from the book."]
    #             # },
    #             {
    #                 "role": m["user"],
    #                 "content": m["User Question: " + user_question + "\n\nRelevant Excerpt(s):\n\n" + relevant_excerpts],
    #             }
    #             for m in st.session_state.messages

    #         ],
    #         model=model,
    #         stream = True
    #     )
        
            #     chat_completion = client.chat.completions.create(
    #         model=model_option,
    #         messages=[
    #             {
    #                 "role": m["role"],
    #                 "content": m["content"]
    #             }
    #             for m in st.session_state.messages
    #         ],
    #         max_tokens=max_tokens,
    #         stream=True
    #     )


        # response = chat_completion.choices[0].message.content
        # return response

    # if user_question := st.text_input("Enter your prompt here..."):
    #     relevant_docs = docsearch.similarity_search(user_question)

    #     relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevant_docs[:3]])        
        
    #     st.session_state.messages.append({"role": "user", "content": user_question})

    #     with st.chat_message("user", avatar='👨‍💻'):
    #         st.markdown(user_question)

    #     # Fetch response from Groq API
    #     try:
    #         # Use the generator function with st.write_stream
    #         with st.chat_message("assistant", avatar="🤖"):
    #             chat_responses_generator = book_chat_completion(client, model, user_question, relevant_excerpts)
    #             full_response = st.write_stream(chat_responses_generator)
    #     except Exception as e:
    #         st.error(e, icon="🚨")

    #     # Append the full response to session_state.messages
    #     if isinstance(full_response, str):
    #         st.session_state.messages.append(
    #             {"role": "assistant", "content": full_response})
    #     else:
    #         # Handle the case where full_response is not a string
    #         combined_response = "\n".join(str(item) for item in full_response)
    #         st.session_state.messages.append(
                
    #             {"role": "assistant", "content": combined_response})
        
    # user_question = st.chat_input("Enter your prompt here...")
    # relevant_docs = docsearch.similarity_search(user_question)

    # relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevant_docs[:3]])
    # response = book_chat_completion(client, model, user_question, relevant_excerpts)
    
        # st.write(full_response)


# if prompt := st.chat_input("Enter your prompt here..."):
    # st.session_state.messages.append({"role": "user", "content": prompt})

    # with st.chat_message("user", avatar='👨‍💻'):
    #     st.markdown(prompt)

    # # Fetch response from Groq API
    # try:
    #     chat_completion = client.chat.completions.create(
    #         model=model_option,
    #         messages=[
    #             {
    #                 "role": m["role"],
    #                 "content": m["content"]
    #             }
    #             for m in st.session_state.messages
    #         ],
    #         max_tokens=max_tokens,
    #         stream=True
    #     )

    #     # Use the generator function with st.write_stream
    #     with st.chat_message("assistant", avatar="🤖"):
    #         chat_responses_generator = generate_chat_responses(chat_completion)
    #         full_response = st.write_stream(chat_responses_generator)
    # except Exception as e:
    #     st.error(e, icon="🚨")

    # # Append the full response to session_state.messages
    # if isinstance(full_response, str):
    #     st.session_state.messages.append(
    #         {"role": "assistant", "content": full_response})
    # else:
    #     # Handle the case where full_response is not a string
    #     combined_response = "\n".join(str(item) for item in full_response)
    #     st.session_state.messages.append(
            
    #         {"role": "assistant", "content": combined_response})