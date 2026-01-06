#!/bin/bash
# Setup script for Raspberry Pi 4

set -e

echo "Setting up Reachy Agent on Raspberry Pi 4..."

# Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    portaudio19-dev \
    git

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository (if needed)
# git clone <repository-url>
# cd reachy-agent

# Setup virtual environment
echo "Creating virtual environment..."
uv venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[voice]"

# Create config
echo "Setting up configuration..."
cp .env.example .env
echo "Edit .env to add your API keys"

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/reachy-agent.service > /dev/null <<EOF
[Unit]
Description=Reachy Agent
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/reachy-agent
Environment="PATH=/home/pi/reachy-agent/.venv/bin"
ExecStart=/home/pi/reachy-agent/.venv/bin/python -m reachy_agent run --voice
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env to add API keys"
echo "2. Test: python -m reachy_agent run --mock"
echo "3. Enable service: sudo systemctl enable reachy-agent"
echo "4. Start service: sudo systemctl start reachy-agent"
