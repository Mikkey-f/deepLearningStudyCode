class Man:
    #构造函数
    def __init__(self, name):
        self.name = name
        print('init')
    def hello(self):
        print('hello')
    def bye(self):
        print('bye')

m = Man('hello')
print(type(m))
m.hello()
m.bye()