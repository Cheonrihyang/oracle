package com.example.spring42.dto;

import com.example.spring42.domain.Article;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
public class ArticleViewResponse {
    private long id;
    private String title;
    private String content;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public ArticleViewResponse(Article article) {
        this.id = article.getId();
        this.title = article.getTitle();
        this.content = article.getContent();
        this.createdAt = article.getCreatedAt();
        this.updatedAt = article.getUpdatedAt();
    }
}
