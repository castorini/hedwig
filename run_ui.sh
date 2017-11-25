#!/usr/bin/env bash

echo "Start the Flask server..."
python3 anserini_dependency/api.py --model idf &
PID_2=$!


echo "Start the JavaScript UI..."
pushd anserini_dependency/js
npm start &
PID_3=$!
popd

# clean up before exiting
function clean_up {
    kill $PID_3
    kill $PID_2
    exit
}

trap clean_up SIGHUP SIGINT SIGTERM SIGKILL
wait
