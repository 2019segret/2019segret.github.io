---
layout: default
title: Python set-up
parent: Posts
nav_order: 2
---

# An advised set-up to start coding in Python
## From a random non-expert programmer

![](img/intro_conda.jpeg)

Read the article on [Medium](https://medium.com/@__initial__/an-advised-set-up-to-start-coding-with-python-5fbb166bd85).

Now, before you begin reading what you came here for, let me remind you that it is my first article (among many I hope) and that you should be tolerant! I will try to make it as simple as possible using a basic stack, constituting a strong basis which you cam build upon in case you become more advanced!

---
### Introduction

Since I will only focus on a set up I assume that you already have an idea of what python is, that you can work your way around a terminal, and will not be giving any code advice. Please be advised that this tutorial works for Unix distributions (MacOs, Linux essentially) but you can easily find the Windows equivalent of my instructions. Now let’s start from the very beginning. To code in Python, you need… Python installed on your computer.

### Let’s dive into it: Download Python

**First Step: Download the Official Installer**

Follow these steps to download the full installer:

Go to the Python.org [Downloads page for macOS](https://www.python.org/downloads/macos/). (or any other OS)
Download whatever macOS 64-bit installer you need. I recommend using either python 3.7 or older since more ancient versions lack dependencies.

**Second Step: Run the Installer**

Run the installer by double-clicking the downloaded file and follow the instructions. Even though it’s probably not useful, note in the back of your head the default location of python i.e where it’s installed.
Congrats you have python installed!

---
### Download Visual Studio Code

Ok, it’s cool you have python but where do you write code now?? To serve that purpose you need a code editor. I strongly recommend using VS code as it is ergonomic, intuitive and if you continue down the coding lane you’re likely to encounter many other coders using that editor. An editor is basically a whole code environment from which you can edit your code, run tests, debug it. A cool thing about VScode is that you can add plug-ins from its marketplace that are very useful.
Same story, head to [Visual Studio Code](https://code.visualstudio.com/download) and choose the proper installer for your OS. Once it’s downloaded, run the installer, follow the instructions and that’s it!
Now open VScode, follow the probable get-to-know instructions and then head straight to the sidebar that should be on the left. Click on “extensions”. The logo looks like this :

![](img/extension.png)

And type in “Python” in the search bar that just opened and install the first choice. This instruction is only to say “Hey VSCode, I’ll be coding in python so provide me with the necessary internal tools to make it work”. You may ask why these tools aren’t already installed, well it’s only to make the whole package lighter from the origin and then you add on what’s necessary to you and you only. It helps to stay consistent with your code too.
Download the IntelliSense one (the first), it should look like this:

![](img/vscode.png)

Now you’re able to run your code!
Let’s go ahead and try. Just before that, check that in the low left corner, in the blue band, a “python interpreter” is selected. Otherwise, click on “Select python interpreter” and a window should open on the top of your window with one option: “Python3.x.x” along with the default path I told you to memorize at the beginning! If not, just select “enter interpreter path” and enter the default path (probably /usr/bin/python3).
Open a new file on VSCode, you should figure out how to do that yourself, and make sure it has the “py” extension! Try typing a simple code line such as print(“Hello World!”) and execute the code with the Run python file logo in the upper right window.

![](img/vscode2.png)

It will display a terminal with what you asked in it. Notice that on the side there is a panel where you can load a folder and see all the files in it. It is mandatory for coding projects!
Congratulations you just ran your first python code on VSCode!
Now just to make things a bit better, also add the “Pylint” extension from the marketplace which is used to write quality code.

---
### Install miniconda if you want to dig a little deeper

You may ask what miniconda is. It’s only a free installer of conda, without any graphic interface. That way you’ll be forced to learn to navigate your way through a computer with your terminal! Now you may ask what conda is. As defined by conda itself, Conda is an open-source package management system for python. It allows installing the packages you need for your code. It allows to easily create virtual environments. You should look at those as little boxes in which you have a full working environment, with the right versions of the necessary packages, and avoid any version issues. You don’t need to know more as you’ll have a “base” environment where you’ll install whatever you need.

*A little interlude*

Since miniconda doesn’t come with a graphic interface you’ll be using your terminal. I recommend using Iterm2 instead of the native terminal if you’re on macOS. It comes with a lot more benefits. I will probably write an article about the way you can properly set up your terminal (mostly to make it look beautiful…). Anyway, let’s get back to business.
You just need to run the two command lines:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda.sh
sh ~/miniconda.sh -b -p $HOME/miniconda3
```

The first one downloads the installer file (bash script) to your home repository (indicated by the ~ symbol). You should divide the line in the following, with wget being a command to get something from the Web.

```
wget link-to-web-page where-to-download
```

The second command line executes the previously downloaded file and stores everything at the path `~/miniconda` (it creates a directory along with it).
There is one last thing to do so that your terminal knows that conda is installed and ready to be used. You need to add conda to your path. “Path” essentially means where the terminal looks to access information. Just run either one of the following commands, depending on your terminal (bash or zsh should be written somewhere at the top of your terminal window). It writes `export PATH=”~/miniconda3/bin:$PATH”` in the file at `~/.xxrc`. This file is a configuration file of your terminal so this is where you tell it your instructions (i.e add conda to its path).

```
#If terminal is using bash 
echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.bashrc

#If terminal is using zsh
echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.zshrc
```

Now run the commands to reload your terminal to make sure it has the updates and to initiate conda :

```
#If terminal is using bash 
bash
conda init

#If terminal is using zsh
zsh
conda init
```

There should be **(base)** written at the beginning of your terminal prompt line. Just like at the beginning of this tutorial, on VSCode you need to link the python associated with conda (remember the whole select interpreter at the bottom left corner…). Just select a new interpreter with the path `~/miniconda3/bin/python`.
You are all set up! I invite you to look at conda instructions but you basically need to know the one line to install packages so that you can import them in your code.

```
conda install name-of-the-package
# If it doesn't work, just run the following
pip3 install name-of-the-package
```

Or look at that well-written [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) to dive a little more into conda.

---

### Conclusion

Thank you guys for reading through that fastidious article that I hope you enjoyed. I tried to make it easy to follow 😊 Please let me know if anything isn’t clear enough. You have my contact address on my profile.
