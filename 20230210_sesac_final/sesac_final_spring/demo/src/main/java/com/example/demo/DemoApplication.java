package com.example.demo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

}


////////////////////////////////////////////////
/////////////// 서버 백엔드 명세서 //////////////

// Maven 으로 구현되었습니다.
// DemoApplication.java 를 실행시키면 localhost:8080 으로 접속이 가능합니다.

// 계획
// main : 로그인 페이지
// userPage : user의 Dash Board 페이지
// detail : 각 이벤트 클릭시 이벤트 로그 및 현장 사진을 볼 수 있는 페이지

// 구현
// main : localhost:8080 접속시 보이는 첫 페이지
// userPage : RDS에 있는 로그데이터 불러와서 볼 수 있음