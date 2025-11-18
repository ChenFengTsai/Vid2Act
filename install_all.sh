#!/bin/bash
set -e  # exit immediately if a command fails

echo "Installing metaworld..."
cd metaworld
pip install -e .
cd ..

echo "Installing rrl-dependencies..."
cd rrl-dependencies
pip install -e .

echo "Installing mj_envs..."
cd mj_envs
pip install -e .
cd ..

echo "Installing mjrl..."
cd mjrl
pip install -e .

echo "✅ All installations completed successfully!"
