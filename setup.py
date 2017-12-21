from setuptools import setup

setup(name='emojify',
      version='0.1',
      description='A python project to overlay an emoji on your face based on the emotion predicted by the convolutional neural network',
      url='http://github.com/storborg/funniest',
      author='Shivam Kumar',
      author_email='shivampkumar@gmail.com',
      license='MIT',
      packages=['emojify'],
      install_requires=[
          'theano',
          'keras',
          'imutils',
          'numpy'
      ],
      zip_safe=False)
