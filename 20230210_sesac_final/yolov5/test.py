import os

path = '/Users/jongya/Desktop/Workspace/lab/20230210_sesac_final/yolov5'
# os.chdir(path)

# import pygame
# pygame.init()
# sound = pygame.mixer.Sound(path + '/beep.wav')
# sound.play()
# pygame.time.wait(int(sound.get_length() * 1000))
# pygame.quit()

from playsound import playsound
playsound(path + '/beep.wav')

# import winsound
# winsound.PlaySound(path + '/beep.wav', winsound.SND_FILENAME)