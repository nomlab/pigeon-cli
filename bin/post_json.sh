#!/bin/sh

curl -X POST -H "Content-Type: application/json" $1 -d @-
