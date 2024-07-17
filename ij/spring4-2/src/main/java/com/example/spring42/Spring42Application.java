package com.example.spring42;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.data.jpa.repository.config.EnableJpaAuditing;

@EnableJpaAuditing	//	JPA 에서 지원하는 audit 기능을 활성화 해주는 어노테이션
					// 자동감지해서 생성일자,수정일자,수정자와 같은 필드관리를 할때 사용
@SpringBootApplication
public class Spring42Application {

	public static void main(String[] args) {
		SpringApplication.run(Spring42Application.class, args);
	}

}
