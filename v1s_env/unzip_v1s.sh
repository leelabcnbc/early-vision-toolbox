#!/usr/bin/env bash

# remove old dir if any
rm -rf v1s-362209feee2e37e3863d6c9ef05a1d8d35899198
unzip v1s-362209feee2e37e3863d6c9ef05a1d8d35899198.zip

#! copy some fixed functions and add-on demo scripts into v1s
cp -f  patched_files/* v1s-362209feee2e37e3863d6c9ef05a1d8d35899198
