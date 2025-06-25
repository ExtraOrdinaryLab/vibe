#!/bin/bash

sudo apt update
sudo apt install tmux sox ffmpeg -y
tmux set-option -g mouse on \; bind -T copy-mode-vi MouseDragEnd1Pane send -X copy-selection-and-cancel

pip install -e .