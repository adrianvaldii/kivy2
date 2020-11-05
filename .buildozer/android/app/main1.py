from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooser

import time
Builder.load_string('''
<Main>:
    BoxLayout:
        Button:
            text:"Use Existing Photo"            
        Button:
            text:"Take New Photo"    
''')

class Main(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")

    def show_file_chooser(self):        
        chooser = self.data_file_dir


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")
   

class FileChooserClick(BoxLayout):    

    def show_file_chooser(self):        
        chooser = self.data_file_dir
    

class CountingIron(App):
    def build(self):
        return Main()
        
CountingIron().run()
