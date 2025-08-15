@echo off
echo ========================================
echo    SMC Strategy - GitHub Setup
echo ========================================
echo.

echo 1. Initializing Git repository...
git init

echo.
echo 2. Adding all files...
git add .

echo.
echo 3. Creating initial commit...
git commit -m "Initial commit: Advanced SMC Trading Strategy v2.5

- Performance: +9.63% in 1.5 months (77% annualized)
- Win Rate: 55.56% with 0.06% max drawdown
- Features: Adaptive risk management, market regime detection
- Includes: Full backtesting suite and analysis tools
- Status: Production ready with comprehensive documentation"

echo.
echo 4. Setting up remote repository...
echo Please create a repository on GitHub first, then run:
echo git remote add origin https://github.com/YOUR_USERNAME/smc-trading-strategy.git
echo git branch -M main
echo git push -u origin main

echo.
echo ========================================
echo Setup complete! Next steps:
echo 1. Create repository on GitHub
echo 2. Add remote origin
echo 3. Push to GitHub
echo ========================================
pause
