// Package webfs embeds the compiled frontend into the Go binary so
// NeuroStudio ships as a single executable.
package webfs

import (
	"embed"
	"io/fs"
)

//go:embed all:files
var embedded embed.FS

// Assets returns the frontend file tree rooted at the web root.
func Assets() fs.FS {
	sub, err := fs.Sub(embedded, "files")
	if err != nil {
		panic(err)
	}
	return sub
}
