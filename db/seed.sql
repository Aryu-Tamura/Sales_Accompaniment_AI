-- seed.sql  (deals テーブルだけを初期投入)
-- 前提: server.py の DDL と同じスキーマ（schedule_json などを含む）
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;

DELETE FROM deals;

-- ========== 商談 (6件) ==========
INSERT INTO deals (deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
VALUES
('DEAL-SHO01','松明化学','田中真奈美','商談',NULL,'[]','2025-08-11','{}','[]','["9/02 初回提案素案共有"]',10,0,NULL),
('DEAL-SHO02','青葉フーズ','渡辺徹','商談',45000000,'[]','2025-08-12','{}','["購買部長"]','["9/03 次回要件定義"]',25,0,NULL),
('DEAL-SHO03','彩光金属','小林恭子','商談',NULL,'[]','2025-08-13','{}','["経営企画"]','[]',10,0,NULL),
('DEAL-SHO04','北斗物流','工藤学','商談',52000000,'[]','2025-08-14','{}','[]','["9/05 価格レンジ打合せ"]',25,0,NULL),
('DEAL-SHO05','旭製材','工藤新一','商談',NULL,'[]','2025-08-16','{}','["CFO"]','[]',10,0,NULL),
('DEAL-SHO06','ミライ精機','田中真奈美','商談',38000000,'[]','2025-08-18','{}','[]','["9/06 評価軸すり合わせ"]',25,0,NULL);

-- ========== 提案・概算条件の提示 (6件) ==========
INSERT INTO deals (deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
VALUES
('DEAL-TEI01','鳳凰テック','工藤学','提案・概算条件の提示',85000000,'[]','2025-08-11','{}','["CFO","経営企画"]','["9/05 概算条件ドラフト提示"]',50,0,NULL),
('DEAL-TEI02','南星商会','工藤新一','提案・概算条件の提示',110000000,'["大型案件"]','2025-08-13','{}','[]','["9/06 比較表共有"]',50,0,NULL),
('DEAL-TEI03','瑞雲製作所','田中真奈美','提案・概算条件の提示',40000000,'[]','2025-08-15','{}','["法務"]','["9/08 条件条項の確認"]',50,0,NULL),
('DEAL-TEI04','栄進食品','渡辺徹','提案・概算条件の提示',95000000,'[]','2025-08-17','{}','[]','["9/09 役員向け説明素案"]',50,0,NULL),
('DEAL-TEI05','暁運輸','小林恭子','提案・概算条件の提示',NULL,'[]','2025-08-20','{}','["購買部長"]','[]',50,0,NULL),
('DEAL-TEI06','白銀化工','工藤学','提案・概算条件の提示',120000000,'["大型案件"]','2025-08-22','{}','[]','["9/10 コスト効果説明"]',50,0,NULL);

-- ========== 申込・審査 (4件) ==========
INSERT INTO deals (deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
VALUES
('DEAL-MOU01','青磁セラミックス','工藤新一','申込・審査',90000000,'["要注意"]','2025-08-11','{"格付":"BBB","LTV":"68%","DSCR":"1.6x"}','["経営企画","法務"]','["9/06 申込書ドラフト","9/12 必要書類回収"]',65,0,NULL),
('DEAL-MOU04','常盤工機','小林恭子','申込・審査',118000000,'["大型案件","要注意"]','2025-08-18','{"格付":"BBB","LTV":"69%","DSCR":"1.5x"}','["法務","購買部長"]','["9/08 財務資料更新"]',65,0,NULL),
('DEAL-MOU05','藤花メディカル','工藤学','申込・審査',47000000,'[]','2025-08-21','{"格付":"BBB","LTV":"66%"}','[]','["9/11 先行レビュー"]',65,0,NULL),
('DEAL-MOU06','蒼天ロジスティクス','田中真奈美','申込・審査',102000000,'["大型案件"]','2025-08-25','{"格付":"BBB+","LTV":"63%","DSCR":"1.7x"}','["CFO"]','["9/13 リスク論点整理"]',65,0,NULL);

-- ========== 稟議書認証 (4件) ==========
INSERT INTO deals (deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
VALUES
('DEAL-RIN01','紅葉電機','渡辺徹','稟議書認証',84000000,'[]','2025-08-12','{"格付":"BBB+","LTV":"64%","DSCR":"1.9x"}','["経営企画"]','["9/09 稟議ドラフト確認"]',80,0,NULL),
('DEAL-RIN02','真砂金属','小林恭子','稟議書認証',132000000,'["大型案件","要注意"]','2025-08-15','{"格付":"BBB","LTV":"67%"}','["法務","CFO"]','["9/10 稟議本申請"]',80,0,NULL),
('DEAL-RIN03','碧海フーズ','工藤学','稟議書認証',56000000,'[]','2025-08-18','{"格付":"BBB+","LTV":"62%"}','[]','["9/11 役員説明資料更新"]',80,0,NULL),
('DEAL-RIN04','黄昏トレード','工藤新一','稟議書認証',101000000,'["大型案件"]','2025-08-22','{"格付":"A-","LTV":"60%","DSCR":"2.0x"}','["CFO"]','["9/12 稟議決裁予定"]',80,0,NULL);

-- ========== 契約手続き (3件) ==========
INSERT INTO deals (deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
VALUES
('DEAL-KEI01','白狼ケミカル','田中真奈美','契約手続き',92000000,'[]','2025-08-13','{"格付":"BBB+","LTV":"61%"}','["法務"]','["9/15 契約書レビュー","9/20 捺印調整"]',95,0,NULL),
('DEAL-KEI02','緋色コンポーネンツ','渡辺徹','契約手続き',205000000,'["大型案件","要注意"]','2025-08-19','{"格付":"A-","LTV":"58%","DSCR":"2.1x"}','["CFO","経理"]','["9/18 日程調整","9/22 クロージング条件確認"]',95,0,NULL),
('DEAL-KEI03','苍月エレクトロ','小林恭子','契約手続き',NULL,'[]','2025-08-25','{"格付":"BBB","LTV":"65%"}','[]','["9/17 捺印体制確認"]',95,0,NULL);

-- ========== 融資実行 (2件) ==========
INSERT INTO deals (deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
VALUES
('DEAL-JIK01','皓星マテリアル','工藤学','融資実行',126000000,'["大型案件"]','2025-08-14','{"格付":"A-","LTV":"57%","DSCR":"2.2x"}','["CFO"]','["9/30 実行日程確定"]',100,0,NULL),
('DEAL-JIK02','常夜情報システム','工藤新一','融資実行',68000000,'[]','2025-08-20','{"格付":"BBB+","LTV":"60%"}','[]','[]',100,0,NULL);

-- ========== 失注 (1件) ==========
INSERT INTO deals (deal_id, company, owner, stage, amount, labels_json, start_date, credit_json, stakeholders_json, schedule_json, progress_pct, is_lost, last_stage_before_lost)
VALUES
('DEAL-LOST1','雲海プロダクツ','田中真奈美','失注',74000000,'["要注意"]','2025-08-21','{"格付":"BBB","LTV":"70%"}','["経営企画"]','["8/25 競合優位判明"]',0,1,'提案・概算条件の提示');

COMMIT;
