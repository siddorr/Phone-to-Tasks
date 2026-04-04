#!/usr/bin/env python3
"""
Phone-to-Tasks Project Readiness Check
"""

import os
import sys
import sqlite3
import importlib
from pathlib import Path
import json

class ReadinessChecker:
    """Check if Phone-to-Tasks is ready for use"""
    
    def __init__(self):
        self.results = {}
        self.issues = []
        self.warnings = []
        
    def check_file_structure(self):
        """Check if all required files exist"""
        print("📁 Checking File Structure...")
        
        required_files = {
            'main.py': 'Main service orchestrator',
            'speech_converter.py': 'Audio to text conversion',
            'call_analyzer.py': 'AI analysis and extraction',
            'translator.py': 'Translation functionality',
            'utils.py': 'Utility functions',
            'config.yaml': 'Configuration file',
            'requirements.txt': 'Python dependencies',
            'demo.py': 'Demo and testing script'
        }
        
        missing_files = []
        existing_files = []
        
        for filename, description in required_files.items():
            if Path(filename).exists():
                size_kb = Path(filename).stat().st_size / 1024
                existing_files.append(f"  ✅ {filename} ({size_kb:.1f} KB) - {description}")
            else:
                missing_files.append(f"  ❌ {filename} - {description}")
        
        for file_info in existing_files:
            print(file_info)
        
        if missing_files:
            for file_info in missing_files:
                print(file_info)
            self.issues.extend(missing_files)
            return False
        
        print(f"✅ All {len(required_files)} core files present")
        return True
    
    def check_dependencies(self):
        """Check Python dependencies"""
        print("\n🔧 Checking Dependencies...")
        
        required_packages = [
            ('openai', 'OpenAI API client'),
            ('yaml', 'YAML configuration parsing'),
            ('sqlite3', 'Database operations'),
            ('json', 'JSON data handling'),
            ('pathlib', 'Path operations'),
            ('logging', 'Logging functionality')
        ]
        
        optional_packages = [
            ('whisper', 'Local Whisper model'),
            ('pydub', 'Audio format conversion'),
            ('pyaudio', 'Audio processing')
        ]
        
        missing_required = []
        missing_optional = []
        
        for package, description in required_packages:
            try:
                importlib.import_module(package)
                print(f"  ✅ {package} - {description}")
            except ImportError:
                print(f"  ❌ {package} - {description} (REQUIRED)")
                missing_required.append(package)
        
        for package, description in optional_packages:
            try:
                importlib.import_module(package)
                print(f"  ✅ {package} - {description}")
            except ImportError:
                print(f"  ⚠️  {package} - {description} (OPTIONAL)")
                missing_optional.append(package)
        
        if missing_required:
            self.issues.append(f"Missing required packages: {', '.join(missing_required)}")
            return False
        
        if missing_optional:
            self.warnings.append(f"Missing optional packages: {', '.join(missing_optional)}")
        
        return True
    
    def check_api_key(self):
        """Check API key configuration"""
        print("\n🔑 Checking API Key...")
        
        try:
            from utils import load_api_key, validate_api_key
            
            api_key = load_api_key()
            
            if not api_key:
                print("  ❌ No API key found")
                self.issues.append("No OpenAI API key configured")
                return False
            
            if api_key == "your-openai-api-key-here":
                print("  ❌ Default placeholder API key")
                self.issues.append("API key is still the default placeholder")
                return False
            
            if validate_api_key(api_key):
                # Don't show the full key for security
                masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "sk-***"
                print(f"  ✅ Valid API key found: {masked_key}")
                return True
            else:
                print("  ⚠️  API key format appears invalid")
                self.warnings.append("API key format may be incorrect")
                return False
                
        except Exception as e:
            print(f"  ❌ API key check failed: {e}")
            self.issues.append(f"API key validation error: {e}")
            return False
    
    def check_database(self):
        """Check database functionality"""
        print("\n💾 Checking Database...")
        
        try:
            # Test database creation and basic operations
            test_db = "readiness_test.db"
            conn = sqlite3.connect(test_db)
            
            # Test table creation
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
            ''')
            
            # Test insert
            conn.execute('INSERT INTO test_table (data) VALUES (?)', ('test_data',))
            conn.commit()
            
            # Test select
            cursor = conn.execute('SELECT * FROM test_table')
            result = cursor.fetchone()
            
            conn.close()
            
            # Clean up test database
            if os.path.exists(test_db):
                os.remove(test_db)
            
            if result and result[1] == 'test_data':
                print("  ✅ Database operations working")
                return True
            else:
                print("  ❌ Database test failed")
                self.issues.append("Database operations not working properly")
                return False
                
        except Exception as e:
            print(f"  ❌ Database error: {e}")
            self.issues.append(f"Database error: {e}")
            return False
    
    def check_existing_data(self):
        """Check if there's existing processed data"""
        print("\n📊 Checking Existing Data...")
        
        db_files = list(Path('.').glob('*.db'))
        
        if not db_files:
            print("  ⚠️  No database files found")
            self.warnings.append("No processed calls found - system is ready but empty")
            return True
        
        for db_file in db_files:
            try:
                conn = sqlite3.connect(str(db_file))
                
                # Check calls table
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='calls'")
                if cursor.fetchone():
                    cursor = conn.execute("SELECT COUNT(*) FROM calls")
                    call_count = cursor.fetchone()[0]
                    
                    # Check action items
                    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='action_items'")
                    if cursor.fetchone():
                        cursor = conn.execute("SELECT COUNT(*) FROM action_items")
                        action_count = cursor.fetchone()[0]
                        print(f"  ✅ {db_file.name}: {call_count} calls, {action_count} action items")
                    else:
                        print(f"  ⚠️  {db_file.name}: {call_count} calls (no action items table)")
                else:
                    print(f"  ⚠️  {db_file.name}: No calls table found")
                
                conn.close()
                
            except Exception as e:
                print(f"  ❌ Error reading {db_file.name}: {e}")
        
        return True
    
    def check_configuration(self):
        """Check configuration file"""
        print("\n⚙️  Checking Configuration...")
        
        if not Path('config.yaml').exists():
            print("  ⚠️  config.yaml not found")
            self.warnings.append("No configuration file - using defaults")
            return True
        
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['audio', 'openai', 'analysis']
            missing_sections = []
            
            for section in required_sections:
                if section in config:
                    print(f"  ✅ {section} section configured")
                else:
                    missing_sections.append(section)
                    print(f"  ⚠️  {section} section missing")
            
            if missing_sections:
                self.warnings.append(f"Config missing sections: {', '.join(missing_sections)}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Config file error: {e}")
            self.issues.append(f"Configuration file error: {e}")
            return False
    
    def test_import_modules(self):
        """Test importing main modules"""
        print("\n🔍 Testing Module Imports...")
        
        modules_to_test = [
            ('utils', 'Utility functions'),
            ('translator', 'Translation functionality'),
            ('speech_converter', 'Speech conversion'),
            ('call_analyzer', 'Call analysis')
        ]
        
        failed_imports = []
        
        for module_name, description in modules_to_test:
            try:
                importlib.import_module(module_name)
                print(f"  ✅ {module_name} - {description}")
            except ImportError as e:
                print(f"  ❌ {module_name} - {description}: {e}")
                failed_imports.append(f"{module_name}: {e}")
        
        if failed_imports:
            self.issues.extend(failed_imports)
            return False
        
        return True
    
    def generate_report(self):
        """Generate final readiness report"""
        
        print("\n" + "="*60)
        print("📋 PHONE-TO-TASKS READINESS REPORT")
        print("="*60)
        
        # Run all checks
        checks = [
            ("File Structure", self.check_file_structure()),
            ("Dependencies", self.check_dependencies()),
            ("API Key", self.check_api_key()),
            ("Database", self.check_database()),
            ("Module Imports", self.test_import_modules()),
            ("Configuration", self.check_configuration()),
            ("Existing Data", self.check_existing_data())
        ]
        
        passed_checks = sum(1 for _, result in checks if result)
        total_checks = len(checks)
        
        print(f"\n🎯 SUMMARY:")
        print(f"  Passed: {passed_checks}/{total_checks} checks")
        
        print(f"\n📊 CHECK RESULTS:")
        for check_name, result in checks:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {check_name:15} {status}")
        
        # Show issues
        if self.issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  • {issue}")
        
        # Show warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Final verdict
        print(f"\n🎯 OVERALL STATUS:")
        if len(self.issues) == 0:
            if len(self.warnings) == 0:
                print("  🟢 FULLY READY - All systems operational!")
                print("\n🚀 Ready to use:")
                print("    python demo.py                    # Test with sample data")
                print("    python process_downloads.py       # Process your audio files")
                print("    python main.py                    # Run main service")
            else:
                print("  🟡 MOSTLY READY - Minor issues to address")
                print("  System can be used but some features may be limited")
        else:
            print("  🔴 NOT READY - Critical issues must be fixed")
            print("\n🔧 Fix these issues first:")
            for i, issue in enumerate(self.issues[:3], 1):  # Show top 3 issues
                print(f"    {i}. {issue}")
        
        return len(self.issues) == 0

def main():
    """Run readiness check"""
    print("🔍 PHONE-TO-TASKS PROJECT READINESS CHECK")
    print("="*50)
    print("Checking system components and configuration...\n")
    
    checker = ReadinessChecker()
    is_ready = checker.generate_report()
    
    return 0 if is_ready else 1

if __name__ == "__main__":
    sys.exit(main())
