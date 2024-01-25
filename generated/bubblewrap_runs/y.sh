
file=$(ls -rt *.mp4 | tail -n 1)
gifski --width 1000 -o z.gif $file 
