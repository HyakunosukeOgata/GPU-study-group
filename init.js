const fs = require('fs');
const path = require('path');

const TOTAL_CHAPTERS = 23;
const MEMBERS = ['MROS', 'fro11o', 'Ogata', 'Jerry'];

function initializeStructure() {

  for (let i = 1; i <= TOTAL_CHAPTERS; i++) {
    const chapter = `ch${i}`;
    
    // 建立章節目錄 (例如: ch1)
    if (!fs.existsSync(chapter)) {
      fs.mkdirSync(chapter, { recursive: true });
    }

    // 建立 [討論.md] 並寫入內容
    const discussionPath = path.join(chapter, '討論.md');
    const discussionContent = `# ${chapter} 討論`;
    fs.writeFileSync(discussionPath, discussionContent, 'utf8');

    // 建立會員子目錄與 [習題.md]
    MEMBERS.forEach(name => {
      const subDirPath = path.join(chapter, name);
      
      if (!fs.existsSync(subDirPath)) {
        fs.mkdirSync(subDirPath, { recursive: true });
      }

      const exercisePath = path.join(subDirPath, '習題.md');
      const exerciseContent = `# ${name} ${chapter} 習題`;
      fs.writeFileSync(exercisePath, exerciseContent, 'utf8');
    });
  }

}

initializeStructure();
