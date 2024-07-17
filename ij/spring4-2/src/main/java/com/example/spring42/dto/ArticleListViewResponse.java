package com.example.spring42.dto;

import com.example.spring42.domain.Article;
import lombok.Getter;

import java.sql.Timestamp;
import java.time.LocalDateTime;

@Getter
public class ArticleListViewResponse {
    private Long id;
    private String title;
    private String content;

    public ArticleListViewResponse(Article article) {
        this.id = article.getId();
        this.title = article.getTitle();
        this.content = article.getContent();
    }
}
