#!/bin/bash

sudo apt update
sudo apt install tmux sox ffmpeg -y
sudo apt install libsox-dev
tmux set-option -g mouse on \; bind -T copy-mode-vi MouseDragEnd1Pane send -X copy-selection-and-cancel

pip install -e .
pip install kaldiio