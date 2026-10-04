CREATE TABLE `sessions` (
	`token` text PRIMARY KEY NOT NULL,
	`tenant` text NOT NULL,
	`role` text NOT NULL,
	`expires` integer NOT NULL
);
--> statement-breakpoint
CREATE TABLE `workspaces` (
	`tenant` text PRIMARY KEY NOT NULL,
	`body` text NOT NULL,
	`version` integer DEFAULT 1 NOT NULL
);
