# Taxbot
RAG based chatbot on taxes, runs on local computer

A RAG chatbot to answer tax questions, developed as a learning project. The bot is a demonstration and can make mistakes or not know certain information. Always double-check with an autoritative source or a human tax professional before making any decisions based on the bot's answers.

The repository had the chatbot code and the original PDFs for the two tax preparers training manuals. The manuals need to be parsed into markdown, chunked, indexed, and stored into a vector database for use in the chatbot. 

Near the bottom of the chatbot file is a line for userID's and passwords. You can take that section out if you are only allowing local access and don't need them. They are there so that you can demonstrate the chatbot on the web, through something like cloudflared.

For more information, see the blog post at 
