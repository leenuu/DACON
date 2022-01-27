import os

class Ai():

    def __init__(self):
        self.main_path = str(os.getcwd())
        self.mode_path = self.main_path + '\\mode'
        self.mode_name_list = os.listdir(self.mode_path)
        self.mode_list = dict()

    def reload(self):
        for mode_file in self.mode_name_list:
            if '.py' in mode_file:
                # exec(f"from mode import {mode_file[:len(mode_file)-3]}")
                # importlib.import_module(f'{mode_file[:len(mode_file)-3]}',[])
                mode_name = f'mode.{mode_file[:len(mode_file)-3]}'
                mod = __import__(f'{mode_name}', fromlist=[mode_name])
                self.mode_list[mode_file[:len(mode_file)-3]] = mod
                print(f'{mode_name} reloaded!')

        print(self.mode_list)
        
    def try_cmd(self, mode, cmd):
        try:
            exec(f"self.mode_list['{mode}'].{cmd}")
        except KeyError:
            print("No command")

Ai = Ai()
# print(bot.mode_name_list)
while True:
    print('Main')
    string = input()

    cmd = string.split(' ')
    print(cmd)

    if cmd[0] == '>Ai':

        if cmd[1] == 'test':
            Ai.try_cmd('base_mode','mode.base_mode_test()')

        elif cmd[1] == 'reload':
            Ai.reload()

        else:
            pass
            
    

    if string == 'exit':
        break
    
    print('')

# test = bot()
# test.reload()
