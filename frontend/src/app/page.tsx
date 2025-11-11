'use client';

import { useState, useEffect } from 'react';
import styles from './page.module.css';

interface Command {
  action: string;
  [key: string]: any;
}

interface ExecutionResult {
  action: string;
  success: boolean;
  result: string;
}

interface CommandResponse {
  commands: Command[];
  string_command?: string;
  execution_results?: ExecutionResult[];
}

// Format model name from snake_case to readable format
// GEMINI_FLASH_2_5 -> Gemini Flash 2.5
// CLAUDE_3_5_SONNET -> Claude 3.5 Sonnet
const formatModelName = (modelName: string): string => {
  return modelName
    .split('_')
    .map((part, index, array) => {
      // If this part is a number and next part is also a number, add a dot after
      if (!isNaN(Number(part)) && index < array.length - 1 && !isNaN(Number(array[index + 1]))) {
        return part + '.';
      }
      // Capitalize first letter of words
      if (isNaN(Number(part))) {
        return part.charAt(0).toUpperCase() + part.slice(1).toLowerCase();
      }
      return part;
    })
    .join(' ')
    .replace(/\s+\./g, '.') // Remove spaces before dots
    .replace(/\.\s+/g, '.'); // Remove spaces after dots
};

export default function Home() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('GEMINI_FLASH_2_5');
  const [commandInput, setCommandInput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<CommandResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [executeMode, setExecuteMode] = useState<boolean>(false);

  // Fetch available models on mount
  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch('/api/models');
      const data = await response.json();
      if (data.models) {
        setModels(data.models);
        if (data.models.includes('GEMINI_FLASH_2_5')) {
          setSelectedModel('GEMINI_FLASH_2_5');
        } else if (data.models.length > 0) {
          setSelectedModel(data.models[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch models:', err);
      setError('Failed to load available models');
    }
  };

  const handleInterpret = async () => {
    if (!commandInput.trim()) {
      setError('Please enter a command');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('/api/interpret', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: commandInput,
          model: selectedModel,
          execute: executeMode,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to interpret command');
      }

      setResults(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel,
          execute: executeMode,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate command');
      }

      setResults(data);
      if (data.string_command) {
        setCommandInput(data.string_command);
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleInterpret();
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.logoContainer}>
          <img src="/rbrgs_logo.ico" alt="RBRGS Logo" className={styles.logo} />
        </div>
        <h1 className={styles.title}>🤖 FRIDA Command Interpreter</h1>
        <p className={styles.subtitle}>Natural Language Robot Command Parser</p>
      </header>

      <main className={styles.main}>
        {/* Controls Panel */}
        <div className={styles.controlPanel}>
          <div className={styles.formGroup}>
            <label htmlFor="model" className={styles.label}>
              Select Model
            </label>
            <select
              id="model"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className={styles.select}
              disabled={loading}
            >
              {models.map((model) => (
                <option key={model} value={model}>
                  {formatModelName(model)}
                </option>
              ))}
            </select>
          </div>

          {/*<div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={executeMode}
                onChange={(e) => setExecuteMode(e.target.checked)}
                className={styles.checkbox}
                disabled={loading}
              />
              Execute commands (simulate execution)
            </label>
          </div>*/}
          
        </div>

        {/* Command Input */}
        <div className={styles.inputSection}>
          <label htmlFor="command" className={styles.label}>
            Enter Command
          </label>
          <textarea
            id="command"
            value={commandInput}
            onChange={(e) => setCommandInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Enter a natural language command... (e.g., 'go to the kitchen and pick up the apple')"
            className={styles.textarea}
            rows={3}
            disabled={loading}
          />
          <div className={styles.hint}>Press Ctrl+Enter to interpret</div>
        </div>

        {/* Action Buttons */}
        <div className={styles.buttonGroup}>
          <button
            onClick={handleInterpret}
            disabled={loading || !commandInput.trim()}
            className={`${styles.button} ${styles.buttonPrimary}`}
          >
            {loading ? '⏳ Processing...' : '🧠 Interpret Command'}
          </button>
          <button
            onClick={handleGenerate}
            disabled={loading}
            className={`${styles.button} ${styles.buttonSecondary}`}
          >
            {loading ? '⏳ Generating...' : '🎲 Generate Random Command'}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className={styles.error}>
            <strong>❌ Error:</strong> {error}
          </div>
        )}

        {/* Results Display */}
        {results && (
          <div className={styles.results}>
            <div className={styles.resultsHeader}>
              <h2>✅ Results</h2>
              {results.string_command && (
                <div className={styles.commandString}>
                  <strong>Original Command:</strong> {results.string_command}
                </div>
              )}
            </div>

            {/* Parsed Commands */}
            {results.commands && results.commands.length > 0 && (
              <div className={styles.commandList}>
                <h3>📋 Parsed Commands:</h3>
                {results.commands.map((cmd, index) => (
                  <div key={index} className={styles.commandCard}>
                    <div className={styles.commandHeader}>
                      <span className={styles.commandIndex}>{index + 1}.</span>
                      <span className={styles.commandAction}>{cmd.action}</span>
                    </div>
                    <div className={styles.commandDetails}>
                      {Object.entries(cmd)
                        .filter(([key]) => key !== 'action')
                        .map(([key, value]) => (
                          <div key={key} className={styles.commandField}>
                            <span className={styles.fieldName}>{key}:</span>
                            <span className={styles.fieldValue}>
                              {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Execution Results */}
            {results.execution_results && results.execution_results.length > 0 && (
              <div className={styles.executionResults}>
                <h3>🚀 Execution Results:</h3>
                {results.execution_results.map((exec, index) => (
                  <div
                    key={index}
                    className={`${styles.executionCard} ${
                      exec.success ? styles.executionSuccess : styles.executionFailure
                    }`}
                  >
                    <div className={styles.executionHeader}>
                      <span className={styles.executionIcon}>
                        {exec.success ? '✅' : '❌'}
                      </span>
                      <span className={styles.executionAction}>{exec.action}</span>
                    </div>
                    <div className={styles.executionResult}>{exec.result}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      <footer className={styles.footer}>
        <div className={styles.footerContent}>
          <p className={styles.footerText}>
            Powered by BAML • Model: {formatModelName(selectedModel)}
          </p>
          <div className={styles.footerLinks}>
            <a
              href="https://github.com/RoBorregos/frida-cortex"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.footerLink}
            >
              GitHub
            </a>
            <span className={styles.footerSeparator}>•</span>
            <a
              href="https://doi.org/10.1007/978-3-032-09037-9_24"
              target="_blank"
              rel="noopener noreferrer"
              className={styles.footerLink}
              title="Taming the LLM: Reliable Task Planning for Robotics Using Parsing and Grounding"
            >
              Paper
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
