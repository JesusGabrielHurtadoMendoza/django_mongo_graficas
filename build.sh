cat > build.sh << 'EOF'
#!/usr/bin/env bash
pip install --upgrade pip
pip install -r requirements.txt
python manage.py collectstatic --noinput
EOF
