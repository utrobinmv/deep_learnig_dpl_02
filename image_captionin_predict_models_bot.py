#!/usr/bin/env python
# pylint: disable=C0116,W0613
# This program is dedicated to the public domain under the CC0 license.
#https://github.com/python-telegram-bot/python-telegram-bot
#https://python-telegram-bot.readthedocs.io/en/stable/telegram.bot.html

"""
First, a few callback functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Example of a bot-user conversation using ConversationHandler.
Send /start to initiate the conversation.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging

import tempfile
import pickle
import io

import torch

import numpy as np
from PIL import Image

from var.token import TOKEN

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackContext,
)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

IMAGE = range(1)

PHOTO, LOCATION, BIO = range(3)

CONST_COMMAND_CANCEL = 'cancel'
CONST_COMMAND_START = 'start'
CONST_COMMAND_SKIP = 'skip'
CONST_COMMAND_HELP = 'help'


from image_captionin_add import VocabTorchText
from image_captionin_add import Decoder
from image_captionin_add import Seq2Seq

from beheaded_inception3 import beheaded_inception_v3

#from skimage.transform import resize

from image_captionin_add import generate_caption

captions = [[]]

vocab_class = VocabTorchText(captions)

vocab_class = pickle.load(open('tmp/vocab_model_multihead.pkl', 'rb'))

print('load model...')

# Model parameters
OUTPUT_DIM = len(vocab_class)
HID_DIM = 128
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
#ENC_PF_DIM = 128

ENC_HIDDEN = 2048

DEC_PF_DIM = 128
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

device = 'cpu'

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

TRG_PAD_IDX = vocab_class.pad_ix

model = Seq2Seq(None, dec, None, TRG_PAD_IDX, device).to(device)

model.load_state_dict(torch.load('tmp/model-multihead-18-epoch.pt', map_location=device))

print('load cv model...')

inception = beheaded_inception_v3().train(False)

print('Ok')
       
def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='JPEG')
    imgByteArr = imgByteArr.getvalue()
    
    return imgByteArr  

def image_resize_width(orig_image, new_width):
    aspect_ratio = orig_image.width / orig_image.height

    new_height = int(new_width / aspect_ratio)

    resized_image = orig_image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

def image_center_crop(im, new_width, new_height):
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))

    return im      
      
class TelegramBotML():
    def __init__(self):
        '''
        Здесь должна быть инициализация бота
        '''
        
    #Action 1
    def _start(self, update: Update, context: CallbackContext) -> int:
        """Starts the conversation and asks the user about their gender."""
        #reply_keyboard = [['Фото', 'Текст','/cancel']]
    
        update.message.reply_text(
            'Привет! Я Image Captioning Bot. Я скажу тебе, что на присылаемых тобою фото. '
            f'Отправь /{CONST_COMMAND_CANCEL} чтобы выйти в начальный режим.\n\n'
            'Просто пришли мне картинку или фото, и я скажу тебе что на ней...',
            reply_markup=ReplyKeyboardRemove(),
        )
    
        return IMAGE        

    
    def _image(self, update: Update, context: CallbackContext) -> int:
        """Stores the image and asks for a location."""
        user = update.message.from_user
 
        photo_file = update.message.document.get_file()
        
        temp_name = next(tempfile._get_candidate_names())
        
        filename = 'tmp/' + temp_name + '.jpg'
        
        photo_file.download(filename)
        
        logger.info("Image of %s: %s", user.first_name, filename)
        update.message.reply_text(
            'Спасибо, теперь дождись результата...'
        )
    
        self._detect_image(filename, update, context)
        
        update.message.reply_text('Работа модели завершена.')
        update.message.reply_text('Просто пришли мне картинку или фото, и я скажу тебе что на ней...')
    
        return IMAGE
    
    def _photo(self, update: Update, context: CallbackContext) -> int:
        """Stores the photo and asks for a location."""
        user = update.message.from_user
        photo_file = update.message.photo[-1].get_file()
        
        temp_name = next(tempfile._get_candidate_names())
        
        filename = 'tmp/' + temp_name + '.jpg'
        
        photo_file.download(filename)
        logger.info("Photo of %s: %s", user.first_name, filename)
        update.message.reply_text(
            'Спасибо, теперь дождись распознавания...!'
        )
    
        self._detect_image(filename, update, context)
        
        update.message.reply_text('Работа модели завершена.')
        update.message.reply_text('Просто пришли мне картинку или фото, и я скажу тебе что на ней...')
    
        return IMAGE
    
    def _detect_image(self, image_filename, update: Update, context: CallbackContext)-> None:
        '''
        Детекция полученной картинки и последующее её удаление
        '''
        logger.info('Здесь будет детекция ' + image_filename)
        
        img = Image.open(image_filename)
        img = image_resize_width(img, 500)
        img = image_center_crop(img, 300, 300)
        img = img.resize((299, 299))
        
        update.message.reply_text('В модель отправленно преобразованное изображение...')
        update.message.reply_photo(image_to_byte_array(img))
        
        img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        img = img.astype('float32') / 255.
                         
        for i in range(10):
            opsi = generate_caption(img, inception, model, vocab_class, t=5.)[1:-1]
            str_text = ' '.join(opsi)                         
            update.message.reply_text(str_text)

    
    def _cancel(self, update: Update, context: CallbackContext) -> int:
        """Cancels and ends the conversation."""
        user = update.message.from_user
        logger.info("User %s canceled the conversation.", user.first_name)
        update.message.reply_text(
            f'Пока! Чтобы начать снова просто введи /{CONST_COMMAND_START}', reply_markup=ReplyKeyboardRemove()
        )
    
        return ConversationHandler.END
    
    def _help_command(self, update: Update, context: CallbackContext) -> None:
        """Send a message when the command /help is issued."""
        update.message.reply_text('Этот бот умеет выполнять разные задания с использованием нейронных сетей. '
                                  'Ты можешь присылать ему картинки или текст, и он будет делать с ними что нибудь интересное!')
        update.message.reply_text(f'Для начала просто введи команду /{CONST_COMMAND_START}')
    
    
    def _echo(self, update: Update, context: CallbackContext) -> None:
        """Echo the user message."""
        update.message.reply_text(f'Возможно мы не поняли твой запрос, попробуй еще раз /{CONST_COMMAND_START} для начала или /{CONST_COMMAND_CANCEL} для отмены!')
        #update.message.reply_text(update.message.text)

        
    def run(self):
        """Run the bot."""
        # Create the Updater and pass it your bot's token.
        updater = Updater(TOKEN)
    
        # Get the dispatcher to register handlers
        dispatcher = updater.dispatcher
    
        # Add conversation handler with the states TYPE_DATA, PHOTO, LOCATION and BIO
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler(CONST_COMMAND_START, self._start)],
            states={
                IMAGE: [MessageHandler(Filters.document.category("image"), self._image),
                        MessageHandler(Filters.photo, self._photo)
                        ],
                
            },
            fallbacks=[CommandHandler(CONST_COMMAND_CANCEL, self._cancel)],
        )
    
        dispatcher.add_handler(conv_handler)
        dispatcher.add_handler(CommandHandler(CONST_COMMAND_HELP, self._help_command))
        dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, self._echo))
    
        # Start the Bot
        updater.start_polling()
    
        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        updater.idle()
       

def main() -> None:
    bot_ml = TelegramBotML()
    bot_ml.run()

if __name__ == '__main__':
    main()
