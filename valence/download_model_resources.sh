#!/bin/bash
#
# Copyright (C) 2020  gizemsogancioglu <gizemsogancioglu@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# Download the model resources from bitbucket repo
DIR="./data/models"
mkdir -p $DIR
wget -O "$DIR/models.zip" https://bitbucket.org/gizemsogancioglu/model-resources/src/055bdebe6dbf69f5ef1ee245b5385dec8c90182d/models.zip

cd $DIR
# Uncompress
unzip models.zip