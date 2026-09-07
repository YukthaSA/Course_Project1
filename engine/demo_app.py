"""
Standalone Local Target Web Application for Playwright Live Testing & Verification
Provides real HTML endpoints for Login, Product Search, Shopping Cart, and Order History.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Demo E-Commerce Portal - Testing Sandbox</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 30px; background: #f4f6f9; }
        .container { max-width: 800px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button { background: #27ae60; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #219150; }
        .alert { padding: 10px; border-radius: 4px; margin-top: 15px; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-danger { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .badge-out-of-stock { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛒 Demo E-Commerce Portal</h1>
        <p>Target Application for Agentic QA Framework Verification</p>
        
        <hr>
        
        <h2>User Authentication</h2>
        <form id="login-form" action="/login" method="POST">
            <div class="form-group">
                <label for="email">Email Address</label>
                <input type="text" id="email" name="email" placeholder="user@example.com" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" id="login-btn">Log In</button>
        </form>

        <hr>

        <h2>Product Catalog Search</h2>
        <div class="form-group">
            <input type="text" id="search-input" placeholder="Search products...">
            <button type="button" id="search-btn" onclick="document.getElementById('search-result').innerText = 'Found 3 items matching search.'">Search</button>
        </div>
        <div id="search-result" style="font-weight: bold; color: #2980b9;"></div>

        <div style="margin-top: 20px;">
            <h3>Product List</h3>
            <ul>
                <li>Wireless Headphones - $99.99 <button type="button" id="add-cart-1">Add to Cart</button></li>
                <li>Gaming Keyboard - $149.99 <span class="badge-out-of-stock">Out of Stock</span></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

class DemoServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        response_html = HTML_PAGE + "<div class='alert alert-success'>✅ Authentication Successful! Welcome back.</div>"
        self.wfile.write(response_html.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Quiet logging

def start_demo_server(port=8080):
    server = HTTPServer(("127.0.0.1", port), DemoServerHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    print(f"🚀 Demo Target Web Application running at http://127.0.0.1:{port}")
    return server

if __name__ == "__main__":
    start_demo_server(8080)
    print("Press Ctrl+C to stop server...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server stopped.")
