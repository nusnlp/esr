#!/bin/bash

xml=$(dirname $BASH_SOURCE)/xml

$(dirname $BASH_SOURCE)/../../script/fews/process.sh $xml
