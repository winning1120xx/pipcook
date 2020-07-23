import fs from 'fs';
import readline from 'readline';

function strip(str: string): string {
    return str.replace(/(^\s*)|(\s*$)/g, '');
}
  

export function MakeWordsSet(words_file: string): Promise<Set<string>> {
    const words_set = new Set<string>();
    const rl = readline.createInterface({
      input: fs.createReadStream(words_file)
    });
    return new Promise((resolve) => {
      rl.on('line', (line: string) => {
        const word = strip(line);
        if (word.length > 0 && !words_set.has(word)) {
          words_set.add(word);
        }
      });
  
      rl.on('close', () => {
        resolve(words_set);
      });
    });
  }