#!/usr/bin/env python3
"""
Demo script for Phone-to-Tasks speech conversion and analysis
"""

import os
import sys
from pathlib import Path
from utils import load_api_key, validate_api_key

def demo_text_analysis():
    """Demo the analysis capabilities with sample text instead of audio"""
    
    # Sample call transcript for testing
    sample_transcript = """
    Hi John, thanks for taking the call. So I wanted to follow up on the project we discussed last week. 
    I think we should schedule a meeting with the team by Friday to review the requirements. 
    Can you send me the documentation by tomorrow? I'll review it and then we can set up that meeting.
    
    Also, I promised to get back to Sarah about the budget numbers. I need to call her before end of week.
    Let me know if Thursday works for you for the team meeting.
    
    One more thing - remember we need to submit the proposal to the client by next Monday. 
    I'll handle the final review, but I'll need your input on the technical sections.
    """
    
    print("🎯 DEMO: Phone-to-Tasks Analysis")
    print("="*50)
    
    try:
        # Import required modules
        from call_analyzer import CallAnalyzer
        
        # Get API key
        api_key = load_api_key()
        if not api_key:
            print("❌ No OpenAI API key found!")
            print("Please either:")
            print("  1. Set environment variable: export OPENAI_API_KEY='your-key'")
            print("  2. Edit api_key.txt file and add your key")
            return
        
        # Initialize analyzer
        analyzer = CallAnalyzer(api_key=api_key)
        
        print("📝 Sample Call Transcript:")
        print("-" * 30)
        print(sample_transcript)
        print("-" * 30)
        
        print("\n🔍 Analyzing...")
        
        # Analyze the sample transcript
        analysis = analyzer.analyze_call(sample_transcript)
        
        if analysis:
            print("\n✅ Analysis Complete!")
            print_analysis_results(analysis)
        else:
            print("❌ Analysis failed")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

def print_analysis_results(analysis):
    """Print formatted analysis results"""
    
    print("\n" + "="*60)
    print("📞 ANALYSIS RESULTS")
    print("="*60)
    
    # Call summary
    summary = analysis.get('summary', {})
    print(f"📋 Call Type: {summary.get('call_type', 'Unknown')}")
    print(f"🎭 Tone: {summary.get('mood_tone', 'Unknown')}")
    print(f"👥 Participants: {', '.join(summary.get('participants', []))}")
    print(f"📊 Overall Confidence: {analysis.get('confidence', 0):.1%}")
    
    # Action items
    actions = analysis.get('action_items', [])
    print(f"\n✅ ACTION ITEMS EXTRACTED ({len(actions)}):")
    print("-" * 40)
    
    for i, action in enumerate(actions, 1):
        print(f"{i}. 📋 {action['description']}")
        print(f"   👤 Who: {action['responsible']}")
        print(f"   ⏰ When: {action.get('deadline', 'No deadline specified')}")
        print(f"   🔥 Priority: {action['priority']}")
        print(f"   📂 Category: {action['category']}")
        print(f"   🎯 Confidence: {action['confidence']:.1%}")
        print()
    
    # Insights summary
    insights = analysis.get('insights', {})
    print("💡 INSIGHTS EXTRACTED:")
    print("-" * 25)
    
    for category, items in insights.items():
        if items and isinstance(items, list):
            print(f"📚 {category.title()}: {len(items)} items")
            
    print("\n" + "="*60)

def main():
    """Main demo function"""
    
    print("🚀 Phone-to-Tasks Demo")
    print("This demo shows speech-to-text conversion and call analysis")
    print()
    
    # Check if we have dependencies
    try:
        import openai
        print("✅ OpenAI library installed")
    except ImportError:
        print("❌ OpenAI library missing")
        print("Install with: pip install openai")
        return
    
    # Check API key
    api_key = load_api_key()
    if api_key:
        print("✅ OpenAI API key found")
    else:
        print("⚠️  No OpenAI API key found")
        print("Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("Or edit api_key.txt file in the project directory")
        print("Continuing with demo using sample data...")
    
    print("\n" + "="*50)
    
    # Run text analysis demo
    demo_text_analysis()
    
    print(f"\n📁 Project files created:")
    print(f"  - main.py (main service)")
    print(f"  - speech_converter.py (audio processing)")  
    print(f"  - call_analyzer.py (AI analysis)")
    print(f"  - requirements.txt (dependencies)")
    print(f"  - config.yaml (configuration)")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'")
    print(f"  2. Install dependencies: pip install -r requirements.txt")
    print(f"  3. Test with audio: python main.py")
    print(f"  4. Add real phone call recordings to process")

if __name__ == "__main__":
    main()
