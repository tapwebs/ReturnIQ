import {sqliteTable,text,integer} from 'drizzle-orm/sqlite-core';
export const workspaces=sqliteTable('workspaces',{tenant:text('tenant').primaryKey(),body:text('body').notNull(),version:integer('version').notNull().default(1)});
export const sessions=sqliteTable('sessions',{token:text('token').primaryKey(),tenant:text('tenant').notNull(),role:text('role').notNull(),expires:integer('expires').notNull()});
