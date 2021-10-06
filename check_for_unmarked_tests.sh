#!/bin/sh

pytest -m "not unit_test and not integration_test" --co > /dev/null 2>&1
if [ $? -eq 5 ]
then
	echo "All tests had either unit_test or integration_test flags."
	exit 0
else
	tput setaf 1; echo "Error: the following tests have neither a unit_test or integration_test mark"
        pytest -m "not unit_test and not integration_test" --co
	exit 1
fi
