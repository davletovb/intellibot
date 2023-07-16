import os
import logging
import re
import asyncio
import aiofiles

from pydub import AudioSegment
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CommandHandler, MessageHandler, filters, CallbackQueryHandler, Application, ContextTypes
from telegram.constants import ChatAction
from cachetools import TTLCache, cached

from prompter import Prompter

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Enable logging for debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Chat history and buffer size
chat_context = TTLCache(maxsize=10, ttl=14400)

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Hi! I am your AI assistant. Ask me anything, and I will try to help!')

# Show typing status while waiting for a response
async def send_typing_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    while True:
        await context.bot.send_chat_action(chat_id=update.message.chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(5)  # Send typing status every 5 seconds

# Process text message
async def process_message(prompter, update, user_message, chat_id):
    url_pattern = r"(https?://\S+)"
    url_match = re.match(url_pattern, user_message)
    if url_match:
        url = url_match.group(1)
        summary = await prompter.save_url(url=url)
        response = "Summary of the web page: " + summary
        await update.message.reply_text(text=response, quote=True)
        user_message = f"{url} saved to my documents database."
    else:
        response = await prompter.generate_response(message=user_message, chat_context=chat_context[chat_id])
        image_url_pattern = r"(https://oaidalleapiprodscus\.blob\..*)"
        image_match = re.match(image_url_pattern, response)
        if image_match:
            image_url = image_match.group(1)
            await update.message.reply_photo(image_url)
        else:
            if update.message.voice or update.message.audio:
                audio = await prompter.generate_audio(text=response)
                if audio:
                    await update.message.reply_voice(voice=audio)
                else:
                    await update.message.reply_text(text=response)
            else:
                await update.message.reply_text(text=response)

    return user_message, response


# Message handler
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    # Get the chat id
    chat_id = update.message.chat_id

    # If the chat context is empty, initialize it
    if chat_id not in chat_context:
        chat_context[chat_id] = []
    
    prompter = Prompter(chat_id=chat_id)
    
    try:
        user_message = None
        # Get the user message
        # Check if the message is a voice message or text message
        if update.message.voice:
            file = await update.message.effective_attachment.get_file()
            await file.download_to_drive("voice_message.ogg")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: AudioSegment.from_ogg("voice_message.ogg").export("voice_message.mp3", format="mp3"))

            with open("voice_message.mp3", "rb") as f:
                transcript = await prompter.transcribe_voice(file=f)
            
            """
            # This async version of this code is not working. Need to fix it.
            async with aiofiles.tempfile.NamedTemporaryFile(prefix="voice_message_", suffix=".ogg") as voice_file:
                file = await update.message.effective_attachment.get_file()
                await file.download_to_drive(voice_file.name)
                await voice_file.flush()

                async with aiofiles.tempfile.NamedTemporaryFile(prefix="voice_message_", suffix=".mp3") as mp3_file:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: AudioSegment.from_ogg(voice_file.name).export(mp3_file.name, format="mp3"))
                    await mp3_file.flush()

                    async with aiofiles.open(mp3_file.name, "rb") as f:
                        transcript = await prompter.transcribe_voice(file=f)
            """
            user_message = transcript

        elif update.message.audio:
            file = await update.message.effective_attachment.get_file()
            await file.download_to_drive("audio_message.mp3")

            with open("audio_message.mp3", "rb") as f:
                transcript = await prompter.transcribe_voice(file=f)
            """
            # This async version of this code is not working. Need to fix it.
            async with aiofiles.tempfile.NamedTemporaryFile(prefix="audio_message_", suffix=".mp3") as audio_file:
                file = await update.message.effective_attachment.get_file()
                await file.download_to_drive(audio_file.name)
                await audio_file.flush()

            async with aiofiles.open(file_name, "rb") as f:
                transcript = await prompter.transcribe_voice(file=f)
            """
            user_message = transcript

        else:
            user_message = update.message.text

        # If the user sends a message, send the message to the chatbot
        if user_message:

            # Create the typing status task
            typing_task = asyncio.create_task(send_typing_status(update, context))

            # Get a response for the user message
            user_message, response = await process_message(prompter, update, user_message, chat_id)

            chat_context[chat_id].extend([{ "Human": user_message, "AI": response }])

            # Stop the typing status task
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        else:
            await update.message.reply_text("Sorry, I don't understand that. Please try again.")

    except Exception as e:
        logger.error(f"Error during message processing: {e}")
        await update.message.reply_text("Sorry, I couldn't process your message. Please try again.")

# Document handler
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get the chat id
    chat_id = update.message.chat_id

    # If the chat context is empty, initialize it
    if chat_id not in chat_context:
        chat_context[chat_id] = []
    
    prompter = Prompter(chat_id=chat_id)
    
    try:
        # Get the document
        file_name = update.message.document.file_name
        file = await update.message.effective_attachment.get_file()
        await file.download_to_drive(file_name)

        # Create the typing status task
        typing_task = asyncio.create_task(send_typing_status(update, context))

        # Save the document to the vector database and get the summary
        summary = await prompter.save_document(document=file_name)

        response = "Summary of the document: " + summary

        await update.message.reply_text(text=response, quote=True)

        chat_context[chat_id].extend([{ "Human": f"{file_name} saved to my documents database.", "AI": response }])

        # Stop the typing status task
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

        os.remove(file_name)
    
    except Exception as e:
        logger.error(f"Error during document processing: {e}")
        await update.message.reply_text("Sorry, I couldn't process your document. Please try again.")


# Clear the document database
async def clear_database(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get the chat id
    chat_id = update.message.chat_id

    prompter = Prompter(chat_id=chat_id)

    # if the database is cleared, send a message to the user
    if await prompter.clear_database():
        await update.message.reply_text(text="Database cleared.")
    else:
        await update.message.reply_text(text="Database not cleared.")


# Select role for the assistant
async def select_role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [
            InlineKeyboardButton("Assistant", callback_data="assistant"),
            InlineKeyboardButton("Teacher", callback_data="teacher"),
            InlineKeyboardButton("Researcher", callback_data="researcher"),
            InlineKeyboardButton("Coder", callback_data="coder"),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Please choose a role:",
                              reply_markup=reply_markup)


async def role_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    role = query.data
    context.user_data["role"] = role
    await query.edit_message_text(text=f"Selected role: {role}")


# Error handler
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:",
                 exc_info=context.error)
    await update.message.reply_text(
        'An error occurred while processing your message. Please try again.')


def main() -> None:
    # Set up the updater and dispatcher
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("selectrole", select_role))
    application.add_handler(CallbackQueryHandler(role_callback))
    application.add_handler(CommandHandler("clear_database", clear_database))
    application.add_handler(MessageHandler(
        filters.TEXT | filters.VOICE | filters.AUDIO & ~filters.COMMAND, message_handler))
    application.add_handler(MessageHandler(
        filters.Document.MimeType("application/pdf") | filters.Document.MimeType("text/plain") | filters.Document.MimeType("application/msword") | filters.Document.MimeType("application/vnd.openxmlformats-officedocument.wordprocessingml.document") | filters.Document.MimeType("text/html") | filters.Document.MimeType("text/csv") | filters.Document.MimeType("text/tab-separated-values") | filters.Document.MimeType("text/richtext"),
        document_handler))
    application.add_error_handler(error_handler)
    # Start the bot
    application.run_polling()


if __name__ == '__main__':
    main()